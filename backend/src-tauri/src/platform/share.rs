//! 产物交付门面：在文件管理器里定位 / 告知落点。
//!
//! 【为什么把"定位产物"收进收敛层】桌面上的交付动作是 `reveal_items_in_dir`
//! （多选高亮）；移动端没有这个动作 —— 系统文件管理器不接受"高亮某几个文件"的
//! 请求，`tauri-plugin-opener` 的 `revealItemInDir` 在移动端是空实现。若不在门面里
//! 分开，移动端用户点"显示导出文件"会得到"成功"但屏幕上什么都没有 —— 这正是本仓
//! 反复禁止的静默失效。
//!
//! 移动端的交付是"**告知落点**"：把文件写进系统可见的文档目录（见
//! [`super::paths::export_dir`]），然后把路径交回界面，由界面说"已保存到 文件 App"。

use std::path::PathBuf;

use tauri_plugin_opener::OpenerExt;

/// 在系统文件管理器中定位一批路径：存在的文件被**多选高亮**，全是目录时打开第一个目录。
///
/// 【为什么走 Rust 而不是前端 `@tauri-apps/plugin-opener`】`opener:default` 实际只授予
/// `open-url` / `reveal-item-in-dir` / `default-urls`，**不含 `open_path`**；而 Rust 侧的
/// `OpenerExt` 调用不过 ACL。
///
/// 【为什么先过滤不存在的项】`reveal_items_in_dir` 内部会 `canonicalize`，任何一个
/// 缺失路径都会让整批 reveal 失败 —— 分轨导出时个别目标被跳过 / 写入失败就会连累
/// 其余文件。
pub fn reveal_or_announce(
    app: &tauri::AppHandle,
    paths: Vec<String>,
    empty_error: &str,
) -> serde_json::Value {
    let mut files: Vec<PathBuf> = Vec::new();
    let mut dirs: Vec<PathBuf> = Vec::new();
    for raw in paths {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let path = PathBuf::from(trimmed);
        if path.is_file() {
            files.push(path);
        } else if path.is_dir() {
            dirs.push(path);
        }
    }

    // ── 移动端：只告知落点 ────────────────────────────────────────────────
    if crate::platform::is_mobile() {
        let anchor = files.first().or_else(|| dirs.first());
        return match anchor {
            Some(path) => {
                let dir = if path.is_dir() {
                    path.clone()
                } else {
                    path.parent()
                        .map(std::path::Path::to_path_buf)
                        .unwrap_or_else(|| path.clone())
                };
                log::info!("[share] delivered to {}", path.display());
                serde_json::json!({
                    "ok": true,
                    "count": files.len(),
                    "path": path.to_string_lossy(),
                    "directory": dir.to_string_lossy(),
                    // `announced` = "没有在文件管理器里高亮，只是告诉了用户位置"。
                    // 前端据此选择文案（"已保存到…" 而不是 "已在文件管理器中显示"）。
                    "announced": true,
                })
            }
            None => serde_json::json!({ "ok": false, "error": empty_error }),
        };
    }

    // ── 桌面：保持重构前的行为 ────────────────────────────────────────────
    if !files.is_empty() {
        return match app.opener().reveal_items_in_dir(files.iter()) {
            Ok(()) => serde_json::json!({ "ok": true, "count": files.len() }),
            Err(error) => {
                // 多选失败（个别文件管理器不支持）→ 至少打开首个文件的所在目录。
                if let Some(parent) = files[0].parent() {
                    let _ = app
                        .opener()
                        .open_path(parent.to_string_lossy(), None::<&str>);
                }
                serde_json::json!({
                    "ok": false,
                    "error": format!("reveal files failed: {error}"),
                })
            }
        };
    }

    if let Some(dir) = dirs.first() {
        return match app.opener().open_path(dir.to_string_lossy(), None::<&str>) {
            Ok(()) => serde_json::json!({ "ok": true, "path": dir.to_string_lossy() }),
            Err(error) => serde_json::json!({
                "ok": false,
                "error": format!("open folder failed: {error}"),
            }),
        };
    }

    serde_json::json!({ "ok": false, "error": empty_error })
}
