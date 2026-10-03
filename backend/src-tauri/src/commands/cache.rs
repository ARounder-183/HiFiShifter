//! 渲染缓存管理命令。
//!
//! 提供前端可调用的统计 / 清理 / 打开目录三个入口。清理与统计都需要扫描缓存
//! 目录并读取文件头，属于低频但可能耗时的 I/O，命令层用 `spawn_blocking`
//! 投递到阻塞线程池执行（见 `commands.rs`），不占用 UI 主线程。
//!
//! 历史背景：本模块曾提供 `clear_cache`（清内存合成缓存 + 删除
//! `<exe_dir>/cache/synth/`），但该目录从未有代码写入，前端也从未调用过。
//! 该命令已随本功能一并移除，避免留下"看起来能清缓存、实际清不掉"的死入口。

use crate::render_cache::{self, ClearScope};
use crate::state::AppState;
use tauri::State;

/// 渲染缓存统计（占用 / 条目 / 分类 / 会话命中率）。
pub(super) fn get_render_cache_stats() -> serde_json::Value {
    let stats = render_cache::stats();
    serde_json::json!({
        "ok": true,
        "enabled": stats.enabled,
        "dir": stats.dir,
        "writable": stats.writable,
        "totalBytes": stats.total_bytes,
        "entries": stats.entries,
        "byKind": stats.by_kind.iter().map(|kind| serde_json::json!({
            "kind": kind.kind,
            "entries": kind.entries,
            "bytes": kind.bytes,
        })).collect::<Vec<_>>(),
        "sessionHits": stats.session_hits,
        "sessionMisses": stats.session_misses,
        "sessionStored": stats.session_stored,
        "sessionWriteErrors": stats.session_write_errors,
        "sessionAccepted": stats.session_accepted,
        "maxSizeBytes": stats.max_size_bytes,
        "maxAgeDays": stats.max_age_days,
    })
}

/// 清理渲染缓存。
///
/// `scope`：`"all"`（默认）/ `"currentProject"` / `"olderThan"` /
/// `"otherSampleRates"`。`days` 仅在 `olderThan` 时有意义（缺省 30 天）。
///
/// 说明：清理只删除磁盘文件，**不动内存缓存** —— 正在播放/等待渲染的片段
/// 仍持有内存 PCM，清缓存不会造成播放中断；下一次真正重渲染时会重新落盘。
pub(super) fn clear_render_cache(
    state: State<'_, AppState>,
    scope: String,
    days: Option<u32>,
) -> Result<serde_json::Value, String> {
    let scope = match scope.as_str() {
        "currentProject" => ClearScope::CurrentProject,
        "olderThan" => ClearScope::OlderThan(u64::from(days.unwrap_or(30))),
        "otherSampleRates" => ClearScope::OtherSampleRates(state.audio_engine.sample_rate_hz()),
        _ => ClearScope::All,
    };
    let report = render_cache::clear(scope);
    Ok(serde_json::json!({
        "ok": true,
        "removedFiles": report.files,
        "removedBytes": report.bytes,
    }))
}

/// 在系统文件管理器中打开渲染缓存目录。
pub(super) fn open_render_cache_dir(app: tauri::AppHandle) -> serde_json::Value {
    use tauri_plugin_opener::OpenerExt;

    let dir = render_cache::current_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return serde_json::json!({
            "ok": false,
            "error": format!("create render cache dir failed: {e}"),
        });
    }
    match app.opener().open_path(dir.to_string_lossy(), None::<&str>) {
        Ok(()) => serde_json::json!({ "ok": true, "path": dir.to_string_lossy() }),
        Err(e) => serde_json::json!({
            "ok": false,
            "error": format!("open render cache dir failed: {e}"),
        }),
    }
}

/// 在系统文件管理器中定位导出音频的产物：优先**选中所有已渲染的文件**（多选），
/// 没有任何现存文件时退化为打开目标文件夹。
///
/// 实现已下沉到 `common::reveal_paths_in_file_manager` —— 文件浏览器右键菜单的
/// 「在文件管理器中显示」要的是同一件事，两处不该各写一遍：过滤缺失项与
/// "多选失败退回打开父目录"的降级策略都是有细节的。这里只负责本场景的空结果文案。
pub(super) fn reveal_export_paths(app: tauri::AppHandle, paths: Vec<String>) -> serde_json::Value {
    crate::commands::common::reveal_paths_in_file_manager(&app, paths, "no export paths found")
}
