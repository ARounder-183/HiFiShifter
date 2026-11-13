//! 系统剪贴板命令：纯文本写入 + HiFiShifter 私有对象格式读写。
//!
//! 【为什么纯文本要单独一条命令】`write_system_clipboard_object` 写的是私有对象
//! 格式 + 一句英文摘要，会把用户剪贴板换成应用内部载荷；而「复制诊断摘要」
//! 「复制路径」「复制仓库链接」要的是能直接贴进 issue 的纯文本。两者共用
//! `hifishifter-clipboard` 的进程锁与争用退避，但格式语义不同。
//!
//! 【为什么走原生】`navigator.clipboard.writeText` 需要 transient user
//! activation 与剪贴板权限；这些菜单动作先动态 import、再发若干 IPC，手势早已
//! 过期，Tauri 的自定义协议也不保证权限。失败却被前端猜成"剪贴板被占用，请重试"
//! —— 重试永远不会好。原生通道无需手势，外部短暂独占由共享层的退避吸收。

use serde_json::{json, Value};

use super::common;

/// 纯文本剪贴板写入：供「复制诊断摘要 / 复制路径 / 复制链接」共用。
#[tauri::command(rename_all = "camelCase")]
pub fn write_clipboard_text(text: String) -> Value {
    common::guard_command(
        "write_clipboard_text",
        json!({ "ok": false, "error": "panic in command: write_clipboard_text" }),
        || {
            if let Err(reason) = crate::system_clipboard::validate_clipboard_text(&text) {
                return json!({ "ok": false, "error": reason });
            }
            match crate::system_clipboard::write_text(&text) {
                Ok(()) => json!({ "ok": true }),
                Err(error) => json!({ "ok": false, "error": error }),
            }
        },
    )
}

#[tauri::command(rename_all = "camelCase")]
pub fn write_system_clipboard_object(payload: String, text_summary: Option<String>) -> Value {
    common::guard_command(
        "write_system_clipboard_object",
        json!({ "ok": false, "error": "panic in command: write_system_clipboard_object" }),
        || {
            let summary = text_summary
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| "HiFiShifter data copied. Paste in HiFiShifter.".to_string());
            // 单剪贴板纪律：本次写入替换整个逻辑剪贴板（时间轴 Clip 载荷随
            // empty() 一并清除），最后复制的获胜。
            match crate::system_clipboard::write_bytes(payload.as_bytes(), &summary) {
                Ok(()) => {
                    // 记录缓存（clipboard_kind 探测与可用性轮询的快速路径）：
                    // 参数线载荷占据槽位，时间轴可用性判定为不可用（counts 0）。
                    if let Some(seq) = crate::system_clipboard::clipboard_seq_num() {
                        crate::system_clipboard::write_clipboard_cache(
                            crate::system_clipboard::ClipboardCacheEntry {
                                seq,
                                hifi_kind: Some("param".to_string()),
                                hifi_clip_count: 0,
                                hifi_track_count: 0,
                                hifi_source_project: None,
                                // write_contents 的 empty() 清除了 REAPERMedia。
                                reaper_available: false,
                            },
                        );
                    }
                    json!({ "ok": true })
                }
                Err(error) => json!({ "ok": false, "error": error }),
            }
        },
    )
}

#[tauri::command(rename_all = "camelCase")]
pub fn read_system_clipboard_object() -> Value {
    common::guard_command(
        "read_system_clipboard_object",
        json!({ "ok": false, "error": "panic in command: read_system_clipboard_object" }),
        || match crate::system_clipboard::read_bytes() {
            Ok(Some(bytes)) => match String::from_utf8(bytes) {
                Ok(text) => json!({ "ok": true, "available": true, "payload": text }),
                // 二进制载荷（如时间轴 MessagePack）对本读取方等价于
                // "没有可用的参数线数据"，由前端回退内部剪贴板。
                Err(_) => json!({ "ok": true, "available": false }),
            },
            Ok(None) => json!({ "ok": true, "available": false }),
            Err(error) => json!({ "ok": false, "error": error }),
        },
    )
}
