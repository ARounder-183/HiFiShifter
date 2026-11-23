//! 工程保存命令入口（从 commands.rs 拆出，控制门面文件行数）。

use tauri::{Manager, State, Window};

use crate::state::AppState;

use super::{common, project};

/// 「另存为」：格式由应用内先选定；format 缺省时保留旧的三过滤器对话框。
#[tauri::command(rename_all = "camelCase")]
pub async fn save_project_as(
    app: tauri::AppHandle,
    window: Window,
    notes_markdown: Option<String>,
    format: Option<String>,
) -> serde_json::Value {
    match tauri::async_runtime::spawn_blocking(move || {
        let state: State<'_, AppState> = app.state();
        project::save_project_as(state, window, format, notes_markdown)
    })
    .await
    {
        Ok(result) => result,
        Err(error) => serde_json::json!({
            "ok": false,
            "error": format!("Failed to join save-as task: {error}"),
        }),
    }
}

/// 只打开原生保存对话框；格式由应用内显式选择。
///
/// 【为什么是 async】与 `commands.rs` 的 dialogs 段同一理由：系统保存对话框是阻塞
/// 调用，同步命令会让 WebView 的 IPC 线程在模态期间完全停摆（桌面症状是"弹着对话框、
/// 状态栏不再刷新"，移动端是死锁）。
#[tauri::command(rename_all = "camelCase")]
pub async fn pick_project_save_path(
    app: tauri::AppHandle,
    format: String,
    default_name: Option<String>,
) -> serde_json::Value {
    tauri::async_runtime::spawn_blocking(move || {
        let state: State<'_, AppState> = app.state();
        common::guard_command(
            "pick_project_save_path",
            serde_json::json!({ "ok": false, "error": "panic in command: pick_project_save_path" }),
            || project::pick_project_save_path(state, format, default_name),
        )
    })
    .await
    .unwrap_or_else(|error| {
        serde_json::json!({ "ok": false, "error": format!("pick_project_save_path task failed: {error}") })
    })
}

/// 保存到指定路径（带版本冲突检测）；force=true 表示用户已确认覆盖。
#[tauri::command(rename_all = "camelCase")]
pub async fn save_project_to_path(
    app: tauri::AppHandle,
    window: Window,
    project_path: String,
    notes_markdown: Option<String>,
    force: Option<bool>,
    format: Option<String>,
) -> serde_json::Value {
    match tauri::async_runtime::spawn_blocking(move || {
        let state: State<'_, AppState> = app.state();
        project::save_project_to_path(state, window, project_path, notes_markdown, force, format)
    })
    .await
    {
        Ok(result) => result,
        Err(error) => serde_json::json!({
            "ok": false,
            "error": format!("Failed to join save task: {error}"),
        }),
    }
}
