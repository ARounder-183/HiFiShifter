use hifishifter_mcp::{EditorBackend, EditorRequest};
use serde_json::{json, Value};
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
};
use tokio_util::sync::CancellationToken;

#[derive(Default)]
struct Backend {
    calls: Mutex<usize>,
}
impl EditorBackend for Backend {
    fn execute(&self, request: EditorRequest) -> Pin<Box<dyn Future<Output = Value> + Send>> {
        *self.calls.lock().unwrap() += 1;
        Box::pin(async move {
            match request {
                EditorRequest::Context => json!({"ok":true,"revision":"visible-project-7"}),
                _ => json!({"ok":false,"code":"REVISION_CONFLICT"}),
            }
        })
    }
}

async fn server() -> (String, Arc<Backend>, CancellationToken) {
    let (listener, address) = hifishifter_mcp::server::bind_local(0).unwrap();
    let url = format!("http://{address}/mcp");
    let backend = Arc::new(Backend::default());
    let stop = CancellationToken::new();
    let backend2 = backend.clone();
    let stop2 = stop.clone();
    tokio::spawn(async move {
        hifishifter_mcp::server::serve(listener, backend2, stop2)
            .await
            .unwrap();
    });
    (url, backend, stop)
}
fn rpc(method: &str, params: Value) -> Value {
    json!({"jsonrpc":"2.0","id":1,"method":method,"params":params})
}

#[test]
fn occupied_configured_port_is_reported_instead_of_silently_using_a_random_port() {
    let (_listener, address) = hifishifter_mcp::server::bind_local(0).unwrap();
    assert!(hifishifter_mcp::server::bind_local(address.port()).is_err());
    assert_eq!(hifishifter_mcp::server::LOCAL_PORT, 39393);
}
async fn request(url: &str, body: Value, session: Option<&str>) -> reqwest::Response {
    let mut request = reqwest::Client::new()
        .post(url)
        .header("Accept", "application/json, text/event-stream")
        .header("MCP-Protocol-Version", "2025-11-25")
        .json(&body);
    if let Some(session) = session {
        request = request.header("Mcp-Session-Id", session);
    }
    request.send().await.unwrap()
}
async fn decode(response: reqwest::Response) -> Value {
    let status = response.status();
    let text = response.text().await.unwrap();
    if let Ok(value) = serde_json::from_str(&text) {
        return value;
    }
    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            if let Ok(value) = serde_json::from_str::<Value>(data.trim()) {
                if value.get("result").is_some() || value.get("error").is_some() {
                    return value;
                }
            }
        }
    }
    panic!("Expected a JSON or SSE JSON-RPC response; HTTP {status}: {text}");
}

#[tokio::test]
async fn local_endpoint_needs_no_authentication_and_rejects_foreign_origins() {
    let (url, backend, stop) = server().await;
    let browser = reqwest::Client::new()
        .post(&url)
        .header("Origin", "https://example.com")
        .json(&rpc("tools/list", json!({})))
        .send()
        .await
        .unwrap();
    assert_eq!(browser.status(), 403);
    assert_eq!(*backend.calls.lock().unwrap(), 0);
    stop.cancel();
}

#[tokio::test]
async fn real_mcp_handshake_lists_typed_tools_and_returns_structured_domain_errors() {
    let (url, backend, stop) = server().await;
    let response=request(&url,rpc("initialize",json!({"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test-agent","version":"1"}})),None).await;
    assert!(response.status().is_success(), "{}", response.status());
    let session = response
        .headers()
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let init = decode(response).await;
    assert!(init["result"].is_object(), "{init}");
    let initialized = json!({"jsonrpc":"2.0","method":"notifications/initialized"});
    let response = request(&url, initialized, session.as_deref()).await;
    assert!(response.status().is_success());
    let listed =
        decode(request(&url, rpc("tools/list", json!({})), session.as_deref()).await).await;
    let tools = listed["result"]["tools"].as_array().unwrap();
    let edit = tools.iter().find(|t| t["name"] == "hfs_edit").unwrap();
    assert!(edit["inputSchema"]
        .to_string()
        .contains("expected_revision"));
    assert!(edit["inputSchema"].to_string().contains("set_static"));
    let context = decode(
        request(
            &url,
            rpc(
                "tools/call",
                json!({"name":"hfs_get_context","arguments":{}}),
            ),
            session.as_deref(),
        )
        .await,
    )
    .await;
    assert_eq!(
        context["result"]["structuredContent"]["revision"],
        "visible-project-7"
    );
    let error=decode(request(&url,rpc("tools/call",json!({"name":"hfs_history","arguments":{"expected_revision":"old","action":"undo"}})),session.as_deref()).await).await;
    assert_eq!(error["result"]["isError"], true);
    assert_eq!(
        error["result"]["structuredContent"]["code"],
        "REVISION_CONFLICT"
    );
    assert_eq!(*backend.calls.lock().unwrap(), 2);
    stop.cancel();
}

#[tokio::test]
async fn invalid_tool_arguments_never_reach_the_editor() {
    let (url, backend, stop) = server().await;
    // Modern stateless calls carry their own metadata. Legacy initialize is tested above.
    let body = rpc(
        "tools/call",
        json!({"name":"hfs_edit","arguments":{"operations":[]},"_meta":{
            "io.modelcontextprotocol/protocolVersion":"2026-07-28",
            "io.modelcontextprotocol/clientInfo":{"name":"test-agent","version":"1"},
            "io.modelcontextprotocol/clientCapabilities":{}
        }}),
    );
    let response = reqwest::Client::new()
        .post(&url)
        .header("Accept", "application/json, text/event-stream")
        .header("MCP-Protocol-Version", "2026-07-28")
        .header("Mcp-Method", "tools/call")
        .header("Mcp-Name", "hfs_edit")
        .json(&body)
        .send()
        .await
        .unwrap();
    let value: Value = response.json().await.unwrap();
    assert!(
        value["error"].is_object() || value["result"]["isError"] == true,
        "{value}"
    );
    assert_eq!(*backend.calls.lock().unwrap(), 0);
    stop.cancel();
}

#[tokio::test]
async fn modern_stateless_client_can_call_the_visible_project_without_initialization() {
    let (url, backend, stop) = server().await;
    let body = rpc(
        "tools/call",
        json!({"name":"hfs_get_context","arguments":{},"_meta":{
            "io.modelcontextprotocol/protocolVersion":"2026-07-28",
            "io.modelcontextprotocol/clientInfo":{"name":"modern-agent","version":"1"},
            "io.modelcontextprotocol/clientCapabilities":{}
        }}),
    );
    let response = reqwest::Client::new()
        .post(&url)
        .header("Accept", "application/json, text/event-stream")
        .header("MCP-Protocol-Version", "2026-07-28")
        .header("Mcp-Method", "tools/call")
        .header("Mcp-Name", "hfs_get_context")
        .json(&body)
        .send()
        .await
        .unwrap();
    let value = decode(response).await;
    assert_eq!(
        value["result"]["structuredContent"]["revision"], "visible-project-7",
        "{value}"
    );
    assert_eq!(*backend.calls.lock().unwrap(), 1);
    stop.cancel();
}
