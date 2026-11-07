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
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

/// 进程内的界面偏好表。
///
/// 【为什么要有它】前端启动时先调 `ui_kv_dump` 把偏好灌回 `localStorage`；
/// 直接回错误会让这一步走失败分支（虽然能降级，但日志里每次开窗都多一条）。
/// CLAP 外壳没有独立的配置文件后端，一份进程内的表至少保证：
/// 同一个会话里开两个窗口看到的是同一份偏好，改语言/外观不会立刻回弹。
///
/// 【边界】它不落盘：宿主退出即丢。真正常驻的偏好由 VST3/App 侧的配置文件负责，
/// 这里只补齐 CLAP 形态缺失的那一段，不假装自己是持久存储。
fn preferences() -> &'static Mutex<BTreeMap<String, String>> {
    static PREFS: OnceLock<Mutex<BTreeMap<String, String>>> = OnceLock::new();
    PREFS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn snapshot() -> Value {
    let map = preferences().lock().unwrap_or_else(|e| e.into_inner());
    json!(map
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect::<serde_json::Map<_, _>>())
}

/// 单个键与整表的硬上限：页面是外部输入，不能让一次 put 把内存吃光。
const MAX_PREF_KEYS: usize = 512;
const MAX_PREF_KEY_BYTES: usize = 256;
const MAX_PREF_VALUE_BYTES: usize = 64 * 1024;

/// 本窗口的 bootstrap：形态是 `capture`，能力全部如实为 false。
///
/// `viewId` 由实例生成，回执与事件都带它（前端据此丢弃别的实例的消息）。
pub fn bootstrap(view_id: &str, capture_caps: Value) -> Value {
    json!({
        "version": 1,
        "viewId": view_id,
        "hostKind": "capture",
        "historyBackend": "local",
        "trackGrouping": true,
        "transportControl": false,
        "fadeAxes": null,
        "captureCaps": capture_caps,
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
        // 【为什么不是 `unbound`】那个取值的前端含义是"视图根本连不上处理器"，而
        // `captureStore` 看到它会**直接跳过 capture_status 轮询** —— 表现就是录制键
        // 永远是灰的（实测：用户报的"当前不可用"）。CLAP 视图确实连着一个处理器：
        // 本模块自己在进程内回答命令。真正缺的是 ARA 文档与编辑能力，那些由
        // `hostCapabilities` 全 false 如实表达。
        "editorRoute": {
            "state": "bound",
            "reason": "CLAP Capture-only shell: commands are served in-process by the plug-in",
        },
    })
}

/// 一条页面请求 → 回执。**永不返回 None**：每个合法请求都必须有回执，否则前端
/// 的 Promise 会挂到 `commandTimeoutMs` 才拒绝，用户看到的是"界面卡住"。
pub fn reply(view_id: &str, request: &Value, capture: &crate::capture::CaptureHost) -> Value {
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
        // 时间线空态：Capture 外壳还没有捕获存储，返回一份**完整**的可渲染空工程。
        // 缺字段会让前端在 reducer 里读到 undefined（例如 bpm / playhead_sec）。
        // 捕获出来的轨道与片段就在这里出现。空时间线（还没录）由运行时给出，
        // 是**一份完整的空工程**而不是省略字段 —— 前端 reducer 读不到 bpm/playhead_sec
        // 会渲染成 undefined。
        "get_timeline_state" | "get_timeline_state_lite" => {
            ok(capture.timeline_payload(request["playheadSec"].as_f64().unwrap_or(0.0)))
        }
        // 只读状态查询：给合法空态，让界面渲染出"还没有内容"，而不是显示一串失败。
        "get_runtime_info" => ok(json!({
            "ok": true,
            "device": "clap",
            "model_loaded": false,
            "audio_loaded": false,
            "has_synthesized": false,
            "is_playing": false,
            "playback_target": null,
            "gpuBackend": "CPU",
        })),
        "get_ui_settings" => ok(json!({})),
        "get_vslib_status" => ok(json!({
            "compiled": false,
            "available": false,
            "version": null,
            "error": "the CLAP Capture-only shell does not load the native vslib",
        })),
        "get_history_state" => ok(json!({
            "ok": true,
            "backend": "local",
            "position": 0,
            "undoDepth": 0,
            "redoDepth": 0,
            "records": [],
        })),
        "plugin_get_apply_state" => ok(json!({
            "generation": 0,
            "applied_generation": 0,
            "pending": false,
            "connected": false,
            "ready": false,
            "error": null,
            "rendering_active": false,
            "render_audible_stale": false,
        })),
        // 捕获命令（与 VST3 同一套名字与载荷）。形状由运行时给出，前端只认一种。
        // 【为什么取 args 而不是整个 request】载荷在 `args` 里；把外层信封传给
        // `capture_delete_take` 会让它读不到 takeId（静默按默认值处理）。
        command if command.starts_with("capture_") => {
            let args = request.get("args").cloned().unwrap_or(Value::Null);
            match capture.handle(command, &args) {
                Ok(value) => ok(value),
                Err(message) => err(&message),
            }
        }
        // 页面就绪/心跳这类诊断消息不进命令表，也别让它们变成错误。
        "plugin_ready" | "plugin_heap" | "plugin_refresh" => ok(json!(null)),
        // 界面偏好：读、写、删都回整表（与后端 `Record<string, string>` 契约一致）。
        "ui_kv_dump" => ok(snapshot()),
        "ui_kv_put" => {
            if let Some(patch) = request["args"]["patch"].as_object() {
                let mut map = preferences().lock().unwrap_or_else(|e| e.into_inner());
                for (key, value) in patch {
                    if map.len() >= MAX_PREF_KEYS && !map.contains_key(key) {
                        continue;
                    }
                    if key.len() > MAX_PREF_KEY_BYTES {
                        continue;
                    }
                    let Some(text) = value.as_str() else { continue };
                    if text.len() > MAX_PREF_VALUE_BYTES {
                        continue;
                    }
                    map.insert(key.clone(), text.to_owned());
                }
            }
            ok(snapshot())
        }
        "ui_kv_delete" => {
            if let Some(keys) = request["args"]["keys"].as_array() {
                let mut map = preferences().lock().unwrap_or_else(|e| e.into_inner());
                for key in keys {
                    if let Some(key) = key.as_str() {
                        map.remove(key);
                    }
                }
            }
            ok(snapshot())
        }
        "save_ui_settings" => ok(json!({ "ok": true })),
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
        let boot = bootstrap(
            "view-1",
            json!({"sawContext": false, "sawInput": false, "sawOffline": false, "canCapture": false}),
        );
        // 【为什么钉死这一条】`editorRoute.state == "unbound"` 会让前端**完全跳过**
        // capture_status 轮询（captureStore.ts 的降频判据），于是录制键永远灰着。
        // 这个取值必须只在"视图真的连不上处理器"时出现，CLAP 不是那种情况。
        assert_ne!(
            boot["editorRoute"]["state"],
            json!("unbound"),
            "an unbound route silences capture polling"
        );
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

    /// 测试用的捕获宿主（未武装，所有命令都回合法空态）。
    fn capture() -> crate::capture::CaptureHost {
        crate::capture::CaptureHost::default()
    }

    #[test]
    fn the_health_probe_succeeds_so_the_app_mounts() {
        let reply = reply(
            "v",
            &json!({ "id": 7, "command": "plugin_health" }),
            &capture(),
        );
        assert_eq!(reply["id"], 7);
        assert_eq!(reply["ok"], true);
        assert_eq!(reply["viewId"], "v");
    }

    #[test]
    fn unwired_commands_fail_fast_instead_of_hanging_until_timeout() {
        let reply = reply(
            "v",
            &json!({ "id": 8, "command": "set_track_state" }),
            &capture(),
        );
        assert_eq!(reply["ok"], false);
        assert_eq!(reply["errorKind"], "transient");
        assert!(reply["error"].as_str().unwrap().contains("not wired"));
    }

    #[test]
    fn every_request_gets_a_reply_even_without_a_command() {
        let reply = reply("v", &json!({ "id": 9 }), &capture());
        assert_eq!(reply["ok"], false);
    }

    #[test]
    fn ui_preferences_round_trip_within_the_session() {
        let put = reply(
            "v",
            &json!({ "id": 1, "command": "ui_kv_put", "args": { "patch": { "hfs.lang": "zh-CN" } } }),
            &capture(),
        );
        assert_eq!(put["ok"], true);
        assert_eq!(put["value"]["hfs.lang"], "zh-CN");
        let dump = reply(
            "v",
            &json!({ "id": 2, "command": "ui_kv_dump" }),
            &capture(),
        );
        assert_eq!(dump["value"]["hfs.lang"], "zh-CN");
        let deleted = reply(
            "v",
            &json!({ "id": 3, "command": "ui_kv_delete", "args": { "keys": ["hfs.lang"] } }),
            &capture(),
        );
        assert!(deleted["value"].get("hfs.lang").is_none());
    }

    #[test]
    fn oversized_or_non_string_preferences_are_rejected_without_erroring() {
        let huge = "x".repeat(MAX_PREF_VALUE_BYTES + 1);
        let reply = reply(
            "v",
            &json!({ "id": 4, "command": "ui_kv_put", "args": { "patch": { "big": huge, "num": 5 } } }),
            &capture(),
        );
        // 非法条目被丢弃，但命令本身成功（与后端"整表回执"的契约一致）。
        assert_eq!(reply["ok"], true);
        assert!(reply["value"].get("big").is_none());
        assert!(reply["value"].get("num").is_none());
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
