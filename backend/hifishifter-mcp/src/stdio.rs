//! Lightweight newline-delimited `stdio` -> Streamable HTTP loopback proxy.
//!
//! Enables stdio-only MCP clients to connect via `HiFiShifter --mcp-stdio`
//! without adding duplicate state or opening a second copy of the project.

use std::io::{BufRead, Write};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

/// Forward newline-delimited JSON-RPC messages from `reader` to `endpoint` over
/// loopback HTTP and write JSON responses to `writer`.
pub async fn proxy_stdio_streams<R, W>(
    endpoint: &str,
    reader: R,
    mut writer: W,
) -> Result<(), String>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    let parsed =
        axum::http::Uri::try_from(endpoint).map_err(|e| format!("invalid endpoint: {e}"))?;
    let host = parsed
        .authority()
        .map(|a| a.as_str().to_string())
        .ok_or_else(|| "endpoint missing host:port authority".to_string())?;
    if !host.starts_with("127.0.0.1:")
        && !host.starts_with("[::1]:")
        && !host.starts_with("localhost:")
    {
        return Err("MCP stdio bridge only connects to loopback endpoints".into());
    }

    let mut lines = tokio::io::BufReader::new(reader).lines();
    let mut session_id: Option<String> = None;

    while let Some(line) = lines.next_line().await.map_err(|e| e.to_string())? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let payload: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(val) => val,
            Err(err) => {
                let err_resp = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": { "code": -32700, "message": format!("Parse error: {err}") }
                });
                let mut out = serde_json::to_string(&err_resp).unwrap_or_default();
                out.push('\n');
                writer
                    .write_all(out.as_bytes())
                    .await
                    .map_err(|e| e.to_string())?;
                writer.flush().await.map_err(|e| e.to_string())?;
                continue;
            }
        };
        let is_notification = payload.get("id").is_none();
        match post_json_rpc(&host, parsed.path(), trimmed, session_id.as_deref()).await {
            Ok((new_session, response_body)) => {
                if let Some(sid) = new_session {
                    session_id = Some(sid);
                }
                if !is_notification && !response_body.trim().is_empty() {
                    let compact = extract_json_rpc_line(&response_body);
                    if !compact.is_empty() {
                        writer
                            .write_all(compact.as_bytes())
                            .await
                            .map_err(|e| e.to_string())?;
                        writer.write_all(b"\n").await.map_err(|e| e.to_string())?;
                        writer.flush().await.map_err(|e| e.to_string())?;
                    }
                }
            }
            Err(err) => {
                if !is_notification {
                    let id = payload
                        .get("id")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    let err_resp = serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32000, "message": format!("HiFiShifter MCP unreachable at {endpoint}: {err}") }
                    });
                    let mut out = serde_json::to_string(&err_resp).unwrap_or_default();
                    out.push('\n');
                    writer
                        .write_all(out.as_bytes())
                        .await
                        .map_err(|e| e.to_string())?;
                    writer.flush().await.map_err(|e| e.to_string())?;
                }
            }
        }
    }
    Ok(())
}

/// Blocking entrypoint suitable for `HiFiShifter --mcp-stdio` CLI invocation.
pub fn run_blocking_stdio_bridge(endpoint: &str) -> Result<(), String> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(async {
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();
        proxy_stdio_streams(endpoint, stdin, stdout).await
    })
}

fn extract_json_rpc_line(raw: &str) -> String {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw.trim()) {
        return serde_json::to_string(&v).unwrap_or_default();
    }
    for line in raw.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data.trim()) {
                if v.get("result").is_some() || v.get("error").is_some() {
                    return serde_json::to_string(&v).unwrap_or_default();
                }
            }
        }
    }
    String::new()
}

/// Minimal dependency-free HTTP/1.1 loopback POST so `hifishifter-mcp` does not
/// pull an HTTP client library into production builds.
async fn post_json_rpc(
    authority: &str,
    path: &str,
    body: &str,
    session_id: Option<&str>,
) -> Result<(Option<String>, String), String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt as _};

    let stream = tokio::net::TcpStream::connect(authority)
        .await
        .map_err(|e| e.to_string())?;
    let (mut rd, mut wr) = stream.into_split();

    let mut req = format!(
        "POST {path} HTTP/1.1\r\nHost: {authority}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nMCP-Protocol-Version: 2025-11-25\r\nConnection: close\r\nContent-Length: {}\r\n",
        body.len()
    );
    if let Some(sid) = session_id {
        req.push_str(&format!("Mcp-Session-Id: {sid}\r\n"));
    }
    req.push_str("\r\n");
    wr.write_all(req.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    wr.write_all(body.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    wr.flush().await.map_err(|e| e.to_string())?;

    let mut buf = Vec::new();
    rd.read_to_end(&mut buf).await.map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&buf);
    let (headers_raw, rest) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response".to_string())?;

    let mut lines = headers_raw.lines();
    let status_line = lines.next().unwrap_or_default();
    let status_code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    let mut returned_session = None;
    let mut chunked = false;
    for header in lines {
        if let Some((k, v)) = header.split_once(':') {
            if k.eq_ignore_ascii_case("mcp-session-id") {
                returned_session = Some(v.trim().to_string());
            } else if k.eq_ignore_ascii_case("transfer-encoding")
                && v.to_ascii_lowercase().contains("chunked")
            {
                chunked = true;
            }
        }
    }

    let decoded_body = if chunked {
        decode_chunked(rest)
    } else {
        rest.to_string()
    };

    if status_code >= 400 {
        return Err(format!("HTTP {status_code}: {decoded_body}"));
    }
    let _ = <std::io::Cursor<&[u8]> as BufRead>::lines;
    let _ = <Vec<u8> as Write>::flush;
    Ok((returned_session, decoded_body))
}

fn decode_chunked(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut pos = 0usize;
    let mut out = Vec::new();
    while pos < bytes.len() {
        let Some(rel_end) = input[pos..].find("\r\n") else {
            break;
        };
        let size_hex = input[pos..pos + rel_end]
            .split(';')
            .next()
            .unwrap_or("")
            .trim();
        let Ok(size) = usize::from_str_radix(size_hex, 16) else {
            break;
        };
        pos += rel_end + 2;
        if size == 0 || pos + size > bytes.len() {
            break;
        }
        out.extend_from_slice(&bytes[pos..pos + size]);
        pos += size + 2;
    }
    String::from_utf8_lossy(&out).into_owned()
}
