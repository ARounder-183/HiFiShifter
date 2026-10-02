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
/// 与「导出布局 / 导出诊断」同一 `tauri_plugin_opener` 通道。`reveal_items_in_dir`
/// 会打开父目录并高亮选中给定文件——Windows 走 `SHOpenFolderAndSelectItems`、
/// macOS 走 Finder 的多选、Linux 走 `FileManager1.ShowItems`，因此分轨导出的多个
/// 文件能一次性全部选中。
pub(super) fn reveal_export_paths(app: tauri::AppHandle, paths: Vec<String>) -> serde_json::Value {
    use tauri_plugin_opener::OpenerExt;

    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    for raw in paths {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let path = std::path::PathBuf::from(trimmed);
        // 只收集真实存在的项：`reveal_items_in_dir` 内部会 canonicalize，
        // 缺失路径会让整批 reveal 失败（分轨时个别目标可能被跳过 / 写入失败）。
        if path.is_file() {
            files.push(path);
        } else if path.is_dir() {
            dirs.push(path);
        }
    }

    if !files.is_empty() {
        return match app.opener().reveal_items_in_dir(files.iter()) {
            Ok(()) => serde_json::json!({ "ok": true, "count": files.len() }),
            Err(e) => {
                // 多选失败（个别文件管理器不支持）→ 至少打开首个文件的所在目录。
                if let Some(parent) = files[0].parent() {
                    let _ = app
                        .opener()
                        .open_path(parent.to_string_lossy(), None::<&str>);
                }
                serde_json::json!({
                    "ok": false,
                    "error": format!("reveal export files failed: {e}"),
                })
            }
        };
    }

    if let Some(dir) = dirs.first() {
        return match app.opener().open_path(dir.to_string_lossy(), None::<&str>) {
            Ok(()) => serde_json::json!({ "ok": true, "path": dir.to_string_lossy() }),
            Err(e) => serde_json::json!({
                "ok": false,
                "error": format!("open export folder failed: {e}"),
            }),
        };
    }

    serde_json::json!({ "ok": false, "error": "no export paths found" })
}
