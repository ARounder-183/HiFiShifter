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

/// 本窗口的 bootstrap：**共享能力快照** + 本格式的路由读数。
///
/// 【为什么能力快照由共享层给】`hostKind` 决定前端按哪种形态分流，`captureCaps` 决定
/// 录制引导语，`hostCapabilities` 决定哪些入口可用 —— 这三样描述的是**插件此刻面对
/// 的宿主**，与外壳是 VST3 还是 CLAP 无关。外壳自己拼一份就必然漂移。
///
/// 【为什么 `editorRoute` 必须在这里补】它是**格式侧**的事实：CLAP 视图连着的是本进程
/// 内的编辑会话。取值为 `unbound` 会让前端**完全跳过** capture_status 轮询
/// （captureStore.ts 的降频判据），于是录制键永远灰着。
pub fn bootstrap(mut capabilities: Value) -> Value {
    capabilities["editorRoute"] = json!({
        "state": "bound",
        "reason": "CLAP shell: commands are served by the shared editor session",
    });
    // 【为什么把 ARA 发现结果交给前端】ARA-over-CLAP 是宿主可选能力：不实现它的宿主
    // （实测 REAPER 7.85）从不查询 ARA 工厂，CLAP 插件便只能是 Capture 形态。界面必须
    // 能如实说清"这个宿主不支持本格式的 ARA，请用 VST3 版"，而不是让用户对着一个
    // 永远不会来的 ARA 会话反复排查。取值只来自宿主是否真的查询过，绝不按名字猜宿主。
    capabilities["araDiscovery"] = json!({
        "factoryQueriedByHost": crate::ara::ara_factory_queried(),
        "format": "clap",
    });
    capabilities
}
/// bootstrap 的 `captureCaps`：四个读数**都是观察到的**。
///
/// 【为什么 sawContext 取自捕获存储而不是 AudioState】“宿主给过可用的工程位置”是捕获
/// 自己的判定（它同时要求采样率有效），而 `AudioState` 只知道“收到过 transport 事件”。
/// 两者混用会让界面在“宿主给了 transport 但位置不可用”时说“能录”。
pub fn capture_caps(saw_context: bool, saw_input: bool, saw_offline: bool, active: bool) -> Value {
    json!({
        "sawContext": saw_context,
        "sawInput": saw_input,
        "sawOffline": saw_offline,
        "canCapture": active || saw_context,
    })
}

/// 一条页面请求 → 回执。
///
/// 【为什么只有一条通路】“命令做什么”只有一套实现：`EditorPort` → `EditorSession` →
/// `commands::dispatch`（Capture 命令在端口里被 `capture::commands::handle` 拦下，与
/// VST3 视图层同一条路）。这一层只做两件事：把页面信封翻成 `(command, args)`，再把
/// 结果包回页面信封。此前这里还有一张“Capture-only 外壳”的命令表，它是同一批命令的
/// 第二份实现 —— 已经删除。
///
/// 【为什么界面偏好不走会话】`ui_kv_*` 是**进程内**的界面偏好表，与会话/文档无关；
/// 它没有共享实现，也不该为它拉起一个编辑会话。
///
/// 【为什么没有端口时如实报错】视图可能还没建好端口（或建端口失败）。那时回一个确定
/// 性的错误，比回一份看起来正常的空状态诚实 —— 后者会让界面把“没有后端”当成“没有内容”。
pub fn reply_with_editor(
    view_id: &str,
    request: &Value,
    editor: Option<&hifishifter_plugin::editor::port::EditorPort>,
) -> Value {
    let id = request.get("id").and_then(Value::as_u64).unwrap_or(0);
    let command = request.get("command").and_then(Value::as_str).unwrap_or("");
    let args = request.get("args").cloned().unwrap_or(Value::Null);
    // 进程内回答的几条：探活与界面偏好。它们必须在**任何**会话之前就能成功，
    // 否则页面在挂载阶段就变成“会话不可用”。
    if let Some(reply) = reply_in_process(view_id, id, command, request) {
        return reply;
    }
    let Some(editor) = editor else {
        return json!({
            "version": 1, "viewId": view_id, "id": id, "ok": false,
            "error": format!("{command}: this view has no editor backend"),
            "errorKind": "transient", "errorCode": null,
        });
    };
    match editor.request(command, args) {
        Ok(value) => json!({ "version": 1, "viewId": view_id, "id": id, "ok": true, "value": value }),
        Err(message) => json!({
            "version": 1, "viewId": view_id, "id": id, "ok": false,
            "error": message, "errorKind": "transient", "errorCode": null,
        }),
    }
}

/// 不依赖编辑会话的那几条命令。返回 `None` 表示“这条要走会话”。
fn reply_in_process(view_id: &str, id: u64, command: &str, request: &Value) -> Option<Value> {
    let ok = |value: Value| {
        Some(json!({ "version": 1, "viewId": view_id, "id": id, "ok": true, "value": value }))
    };
    match command {
        // 挂载前的探活（sessionProbe.ts）：必须成功，否则整页变成“会话不可用”。
        "plugin_health" | "ping" => ok(json!({ "ok": true, "kind": "clap" })),
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
        "save_ui_settings" | "get_ui_settings" => ok(json!({ "ok": true })),
        // 页面就绪/心跳这类诊断消息不进命令表，也别让它们变成错误。
        "plugin_ready" | "plugin_heap" | "plugin_refresh" => ok(json!(null)),
        // 【为什么这两条在进程内回答】它们服务的是**这个视图**，与编辑会话无关：
        // `log_frontend_error` 只是把前端错误按共享限流键写进日志（与 VST3 同一个
        // 实现）；`host_key_owned_set` 是 VST3 让路状态机的输入，而 CLAP 没有
        // `IPlugView::keyDown` 那条路（键直达 WebView），让路无从发生。此前它们落到
        // actor 上变成 "Command unavailable in plug-in mode"，每次开窗都往日志里
        // 留下两条假错误。
        "log_frontend_error" => {
            hifishifter_plugin::editor::log_frontend_error(&request["args"]);
            ok(json!(null))
        }
        "host_key_owned_set" => ok(json!({ "ok": true, "hostRelay": false })),
        "" => Some(json!({
            "version": 1, "viewId": view_id, "id": id, "ok": false,
            "error": "missing command", "errorKind": "transient", "errorCode": null,
        })),
        _ => None,
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
        let boot = bootstrap(json!({
            "version": 1,
            "viewId": "view-1",
            "hostKind": "capture",
            "hostCapabilities": {
                "transport": false, "clipGeometry": false, "clipWrite": false, "split": false,
                "undoStack": false, "mediaImport": false, "trackCreate": false,
                "itemClipboard": false, "fadeAxes": false, "exportAudio": false, "exportData": false
            },
            "captureCaps": {"sawContext": false, "sawInput": false, "sawOffline": false, "canCapture": false}
        }));
        // 【为什么钉死这一条】`editorRoute.state == "unbound"` 会让前端**完全跳过**
        // capture_status 轮询（captureStore.ts 的降频判据），于是录制键永远灰着。
        // 这个取值必须只在“视图真的连不上处理器”时出现，CLAP 不是那种情况。
        assert_ne!(
            boot["editorRoute"]["state"],
            json!("unbound"),
            "an unbound route silences capture polling"
        );
        assert_eq!(boot["version"], 1);
        assert_eq!(boot["viewId"], "view-1");
        assert_eq!(boot["hostKind"], "capture");
        // ARA 发现结果必须如实来自宿主是否查询过工厂，且标明格式。
        assert_eq!(boot["araDiscovery"]["format"], json!("clap"));
        assert_eq!(
            boot["araDiscovery"]["factoryQueriedByHost"],
            json!(crate::ara::ara_factory_queried())
        );
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

    /// `captureCaps` 的四个读数必须与**观察到的**事实一致。
    #[test]
    fn capture_caps_mirror_the_observed_flags() {
        let caps = capture_caps(true, false, true, false);
        assert_eq!(caps["sawContext"], true);
        assert_eq!(caps["sawInput"], false);
        assert_eq!(caps["sawOffline"], true);
        assert_eq!(caps["canCapture"], true);
        // 激活过（但还没见过位置）也能先武装 —— 否则停播时宿主一个块都不送，
        // 按钮在用户做任何事之前永远是灰的。
        assert_eq!(capture_caps(false, false, false, true)["canCapture"], true);
        assert_eq!(capture_caps(false, false, false, false)["canCapture"], false);
    }

    /// **探活必须成功**：否则页面在挂载阶段就变成“会话不可用”。
    #[test]
    fn the_health_probe_succeeds_so_the_app_mounts() {
        let reply = reply_with_editor("v", &json!({ "id": 7, "command": "plugin_health" }), None);
        assert_eq!(reply["id"], 7);
        assert_eq!(reply["ok"], true);
        assert_eq!(reply["viewId"], "v");
    }

    /// **没有编辑后端的视图必须如实报错**，而不是回一份看起来正常的空状态。
    ///
    /// 【为什么这条必须存在】把“没有后端”伪装成“没有内容”，用户会以为是工程空了；
    /// 而错误必须**立刻**回来 —— 挂到 30 s 超时是另一种不可用。
    #[test]
    fn a_session_backed_command_without_a_port_fails_fast_instead_of_hanging() {
        for command in ["get_timeline_state", "capture_status", "get_param_frames"] {
            let reply = reply_with_editor("v", &json!({ "id": 3, "command": command }), None);
            assert_eq!(reply["ok"], false, "{command} must not fake a value");
            assert_eq!(reply["errorKind"], "transient");
            assert!(
                reply["error"].as_str().unwrap_or_default().contains("no editor backend"),
                "the error must name the real cause: {reply}"
            );
        }
    }

    /// **捕获命令走的是共享处理函数**：`capture_status` 由 `capture::commands::handle`
    /// 回答（与 VST3 视图层同一条路），不是本文件里的第二份实现。
    #[test]
    fn capture_commands_are_served_by_the_shared_handler() {
        let owner = std::sync::Arc::new(
            hifishifter_plugin::render::extension::ExtensionOwner::default(),
        );
        let port = hifishifter_plugin::editor::port::EditorPort::new(owner, "v");
        let reply = reply_with_editor(
            "v",
            &json!({ "id": 5, "command": "capture_status" }),
            Some(&port),
        );
        assert_eq!(reply["ok"], true, "{reply}");
        // 载荷形状由运行时的 `status::status_json` 给出，与 VST3 逐字段相同。
        assert_eq!(reply["value"]["state"], "empty", "{reply}");
        assert_eq!(reply["value"]["canCapture"], false, "{reply}");
    }

    #[test]
    fn every_request_gets_a_reply_even_without_a_command() {
        let reply = reply_with_editor("v", &json!({ "id": 9 }), None);
        assert_eq!(reply["ok"], false);
    }

    #[test]
    fn ui_preferences_round_trip_within_the_session() {
        let put = reply_with_editor(
            "v",
            &json!({ "id": 1, "command": "ui_kv_put", "args": { "patch": { "hfs.lang": "zh-CN" } } }),
            None,
        );
        assert_eq!(put["ok"], true);
        assert_eq!(put["value"]["hfs.lang"], "zh-CN");
        let dump = reply_with_editor("v", &json!({ "id": 2, "command": "ui_kv_dump" }), None);
        assert_eq!(dump["value"]["hfs.lang"], "zh-CN");
        let deleted = reply_with_editor(
            "v",
            &json!({ "id": 3, "command": "ui_kv_delete", "args": { "keys": ["hfs.lang"] } }),
            None,
        );
        assert!(deleted["value"].get("hfs.lang").is_none());
    }

    #[test]
    fn oversized_or_non_string_preferences_are_rejected_without_erroring() {
        let huge = "x".repeat(MAX_PREF_VALUE_BYTES + 1);
        let reply = reply_with_editor(
            "v",
            &json!({ "id": 4, "command": "ui_kv_put", "args": { "patch": { "big": huge, "num": 5 } } }),
            None,
        );
        // 非法条目被丢弃，但命令本身成功（与后端“整表回执”的契约一致）。
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
