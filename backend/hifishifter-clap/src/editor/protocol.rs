//! CLAP 编辑器窗口的页面协议：bootstrap 与命令回执（纯函数，可全量单测）。
//!
//! 【为什么不复用 VST3 的 `EditorClient`】它握着 `EditorLink` / `UiSink` /
//! ARA 文档租约，219 条命令的每一次分发都要问会话状态。CLAP 阶段 1 是
//! Capture-only 外壳，没有 ARA 文档，硬接那套只会把"窗口能开、每条命令超时"
//! 变成新的白屏。这里给一份**如实**的最小后端：探活成功、只读空态给得出、
//! 未接线的命令立刻回一个带分类的错误，前端不会挂到 30 s。
//!
//! 【为什么这不是假页面】前端资源、bootstrap、消息口三样都是真的；页面会完整
//! 挂载同一份 React 应用。能力边界由 `hostCapabilities` 如实表达（全是 false），
//! 用户看到的是"这个形态还没有编辑后端"，而不是一块空白或永远转的加载圈。

use serde_json::{json, Value};

/// 本窗口的 bootstrap：形态是 `capture`，能力全部如实为 false。
///
/// `viewId` 由实例生成，回执与事件都带它（前端据此丢弃别的实例的消息）。
pub fn bootstrap(view_id: &str) -> Value {
    json!({
        "version": 1,
        "viewId": view_id,
        "hostKind": "capture",
        "historyBackend": "local",
        "trackGrouping": true,
        "transportControl": false,
        "fadeAxes": null,
        "captureCaps": {
            "sawContext": false,
            "sawInput": false,
            "sawOffline": false,
            "canCapture": false,
        },
        "hostCapabilities": {
            "transport": false,
            "clipGeometry": false,
            "clipWrite": false,
            "split": false,
            "undoStack": false,
            "mediaImport": false,
            "trackCreate": false,
            "itemClipboard": false,
            "fadeAxes": false,
            "exportAudio": false,
            "exportData": false,
        },
        "editorRoute": {
            "state": "unbound",
            "reason": "CLAP Capture-only shell: the shared ARA editor is not attached",
        },
    })
}

/// 一条页面请求 → 回执。**永不返回 None**：每个合法请求都必须有回执，否则前端
/// 的 Promise 会挂到 `commandTimeoutMs` 才拒绝，用户看到的是"界面卡住"。
pub fn reply(view_id: &str, request: &Value) -> Value {
    let id = request.get("id").and_then(Value::as_u64).unwrap_or(0);
    let command = request.get("command").and_then(Value::as_str).unwrap_or("");
    let ok = |value: Value| json!({ "version": 1, "viewId": view_id, "id": id, "ok": true, "value": value });
    let err = |message: &str| {
        json!({
            "version": 1, "viewId": view_id, "id": id, "ok": false,
            "error": message, "errorKind": "transient", "errorCode": null,
        })
    };
    match command {
        // 挂载前的探活（sessionProbe.ts）：必须成功，否则整页变成"会话不可用"。
        "plugin_health" | "ping" => ok(json!({ "ok": true, "kind": "clap-capture" })),
        // 时间线空态：Capture 外壳还没有捕获存储，返回一份可渲染的空工程。
        "get_timeline_state" => ok(json!({ "tracks": [], "clips": [] })),
        // 页面就绪/心跳这类诊断消息不进命令表，也别让它们变成错误。
        "plugin_ready" | "plugin_heap" | "plugin_refresh" => ok(json!(null)),
        // UI 偏好落盘：阶段 1 用 WebView2 自己的 localStorage 兜底，明确告知而非超时。
        "save_ui_settings" => ok(json!(null)),
        "" => err("missing command"),
        _ => err("command is not wired in the CLAP Capture-only shell"),
    }
}

/// 校验一条页面消息是否属于本窗口；不合法就不回执（协议层保持安静）。
pub fn parse_request(text: &str, view_id: &str) -> Option<Value> {
    if text.len() > 1024 * 1024 {
        return None;
    }
    let request: Value = serde_json::from_str(text).ok()?;
    if request.get("version").and_then(Value::as_u64) != Some(1)
        || request.get("viewId").and_then(Value::as_str) != Some(view_id)
    {
        return None;
    }
    request.get("id").and_then(Value::as_u64)?;
    Some(request)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_reports_capture_and_no_host_write_capabilities() {
        let boot = bootstrap("view-1");
        assert_eq!(boot["version"], 1);
        assert_eq!(boot["viewId"], "view-1");
        assert_eq!(boot["hostKind"], "capture");
        let caps = &boot["hostCapabilities"];
        for key in [
            "transport",
            "clipGeometry",
            "clipWrite",
            "split",
            "undoStack",
            "mediaImport",
            "trackCreate",
            "itemClipboard",
            "fadeAxes",
            "exportAudio",
            "exportData",
        ] {
            assert_eq!(caps[key], false, "{key} must be false in stage 1");
        }
        assert_eq!(boot["captureCaps"]["canCapture"], false);
    }

    #[test]
    fn the_health_probe_succeeds_so_the_app_mounts() {
        let reply = reply("v", &json!({ "id": 7, "command": "plugin_health" }));
        assert_eq!(reply["id"], 7);
        assert_eq!(reply["ok"], true);
        assert_eq!(reply["viewId"], "v");
    }

    #[test]
    fn unwired_commands_fail_fast_instead_of_hanging_until_timeout() {
        let reply = reply("v", &json!({ "id": 8, "command": "set_track_state" }));
        assert_eq!(reply["ok"], false);
        assert_eq!(reply["errorKind"], "transient");
        assert!(reply["error"].as_str().unwrap().contains("not wired"));
    }

    #[test]
    fn every_request_gets_a_reply_even_without_a_command() {
        let reply = reply("v", &json!({ "id": 9 }));
        assert_eq!(reply["ok"], false);
    }

    #[test]
    fn foreign_view_messages_are_ignored() {
        assert!(parse_request(
            r#"{"version":1,"viewId":"other","id":1,"command":"ping"}"#,
            "v"
        )
        .is_none());
        assert!(
            parse_request(r#"{"version":1,"viewId":"v","id":1,"command":"ping"}"#, "v").is_some()
        );
        assert!(parse_request("not json", "v").is_none());
    }
}
